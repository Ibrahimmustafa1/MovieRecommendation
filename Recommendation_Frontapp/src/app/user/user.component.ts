import { Component } from '@angular/core';
import { ApiService } from '../api.service';
import { OwlOptions } from 'ngx-owl-carousel-o';

@Component({
  selector: 'app-user',
  templateUrl: './user.component.html',
  styleUrls: ['./user.component.css']
})
export class UserComponent {
    selectes_user: any;
    loading: boolean = false;
    numbers: number[] = [];
    movies =[]
    high_rates: any[] = [];
    low_rates: any[] = [];
    movies_details:any = [];
    high_movie_rec_details:any = [];
    low_movie_rec_details:any = [];
    high_rated_movies = [];
    low_rated_movies = [];
    recommendations_movies = [];
    imgpre:string='https://image.tmdb.org/t/p/w500/';
    user_id :any;
    customOptions: OwlOptions = {
        margin:10,
        loop: true,
        mouseDrag: true,
        touchDrag: true,
        pullDrag: false,
        dots: false,
        navSpeed: 700,
        navText: ['', ''],
        responsive: {
          0: {
            items: 2
          },
          400: {
            items: 4
          },
          740: {
            items: 3
          },
          940: {
            items: 6
          }
        },
        nav: true
      }
    constructor(private apiService: ApiService) {

      for (let i = 1; i <= 611; i++) {
        this.numbers.push(i);
      }
    }
  
    selectNumber(number: number) {
        this.selectes_user = number;
        this.loading = false;
        this.user_id = number;
        this.apiService.getUserMovie(number).subscribe((data: any) => {
            data = data.map((movie: any) => {
                return {
                    tmdbId: movie.tmdbId,
                };
            }
            );
            this.movies = data;
            this.getMovieDetails();
        });
        this.getMovieDetails();
     
    }

    getMovieDetails() {
        this.movies_details = [];
        for (let movie of this.movies) {
            this.apiService.getMovieDetails(movie['tmdbId']).subscribe((data: any) => {
                this.movies_details.push(data);
            });
            console.log(this.movies_details);
        }
        this.getRecommendations();
    }

    getRecommendations() {

        this.apiService.getMovieRecommendations(this.user_id).subscribe((data: any) => {
            this.high_rated_movies = data['high'];
            this.low_rated_movies = data['low'];
            console.log(this.high_rated_movies);
            console.log("===========")
            console.log(this.low_rated_movies);
            this.getRecommendationsMovieDetails(this.high_rated_movies,'high');
            this.getRecommendationsMovieDetails(this.low_rated_movies,'low');
        });
    }


 getRecommendationsMovieDetails(movies:any,label:string) {
    
        if (label == 'high') {
            this.high_movie_rec_details = [];
            for (let movie of movies) {
                
                this.apiService.getMovieDetails(movie['tmdb_id']).subscribe((data: any) => {
                    data['rate'] = movie['rate'];
                    this.high_movie_rec_details.push(data);
                   
                });
            }
        }
        else {

            
            for (let movie of movies) {
                this.low_movie_rec_details = [];

                this.low_rates.push(movie['rate'])
                this.apiService.getMovieDetails(movie['tmdb_id']).subscribe((data: any) => {
                    data['rate'] = movie['rate'];
                    this.low_movie_rec_details.push(data);
                });
            }
        }
        this.loading = true;
    }

}
